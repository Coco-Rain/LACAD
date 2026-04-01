import cadquery as cq

result = (
cq.Workplane('XY')
.rect(10.0, 10.0)
.extrude(2.0)
.faces('>Z')
.circle(2.0)
.extrude(5.0)
.faces('<Z[1]')
.edges(cq.selectors.NearestToPointSelector((0.0, 0.0)))
.fillet(0.1)
)
cq.exporters.export(result, 'GT.stl')