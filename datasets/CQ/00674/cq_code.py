import cadquery as cq

result = (
cq.Workplane("XY" )
.circle(100)
.extrude(130)
.faces(">Z")
.workplane(centerOption="CenterOfMass")
.transformed(rotate=(0, 0, 90), offset=(0, 0, -50))
.transformed(rotate=(90, 0, 0), offset=(40, 15, 0))
.circle(20)
.cutBlind("last")
.faces(">Z")
.workplane(offset=-50)
.split(keepBottom=True)
)
cq.exporters.export(result, 'GT.stl')