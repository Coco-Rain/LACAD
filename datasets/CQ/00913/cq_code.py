import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(20, 20)
.rarray(5, 5, 3, 3)
.circle(1)
.finalize()
)
extruded_result = result.extrude(5)
cq.exporters.export(result, 'GT.stl')